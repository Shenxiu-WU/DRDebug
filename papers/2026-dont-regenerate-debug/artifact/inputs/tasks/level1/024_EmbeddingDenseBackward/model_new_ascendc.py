import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _embedding_dense_backward_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False):
        assert grad_output.dim() >= 2
        assert indices.dim() >= 1
        assert grad_output.shape[-1] > 0
        assert grad_output.is_contiguous()
        assert indices.is_contiguous()
        assert grad_output.dtype in (torch.float32, torch.float16, torch.bfloat16)

        original_shape = list(grad_output.shape)
        D = original_shape[-1]

        # Flatten to 2D / 1D
        grad_output_2d = grad_output.reshape(-1, D).contiguous()
        indices_1d = indices.reshape(-1).contiguous().to(torch.int32)

        grad_weight, = _ext.run_embedding_dense_backward(
            grad_output_2d,
            indices_1d,
            num_weights,
            padding_idx,
            scale_grad_by_freq,
        )

        return grad_weight
