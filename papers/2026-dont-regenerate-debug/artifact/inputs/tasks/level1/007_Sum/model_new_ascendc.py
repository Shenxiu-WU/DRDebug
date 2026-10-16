"""AscendC-optimised implementation of torch.sum(input, dim, keepdim).

The heavy lifting is delegated to an AscendC kernel that reduces the last
dimension of a 2D tensor.  This wrapper handles:
  - dim normalisation (negative indices, dim=None, tuple dims)
  - reshaping the input so the reduction axis is always the last one
  - calling the AscendC kernel via pybind11 extension
  - squeezing / reshaping the output to match the expected shape
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _sum_ext as _ext


class ModelNew(nn.Module):
    """Drop-in replacement for torch.sum using an AscendC reduction kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        # --- Edge case: empty tensor ---
        if x.numel() == 0:
            if dim is None:
                if keepdim:
                    return torch.zeros([1] * x.ndim, dtype=x.dtype, device=x.device)
                return torch.zeros([], dtype=x.dtype, device=x.device)
            if isinstance(dim, int):
                dim = [dim]
            ndim = x.ndim
            dim = [d % ndim for d in dim]
            out_shape = list(x.shape)
            for d in dim:
                if keepdim:
                    out_shape[d] = 1
                else:
                    out_shape[d] = -1
            if not keepdim:
                out_shape = [s for s in out_shape if s != -1]
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # --- dim=None -> total reduction ---
        if dim is None:
            orig_shape = x.shape
            N = x.numel()
            x_2d = x.reshape(1, N).contiguous()
            result_2d = _ext.run_sum(x_2d)  # (1, 1)
            result = result_2d.squeeze(0)  # (1,)
            if not keepdim:
                result = result.squeeze(0)  # scalar ()
            else:
                result = result.reshape([1] * len(orig_shape))
            return result

        # --- Normalise dim to a list of positive ints ---
        if isinstance(dim, int):
            dim = [dim]
        ndim = x.ndim
        dim = [d % ndim for d in dim]

        # --- Compute input reshape: move reduction dims to the end ---
        reduce_dims = sorted(set(dim))
        other_dims = [i for i in range(ndim) if i not in reduce_dims]

        # New permutation: [other_dims..., reduce_dims...]
        perm = other_dims + reduce_dims
        x_perm = x.permute(perm).contiguous()

        # Flatten leading (non-reduce) dims into M, trailing (reduce) dims into N
        M = 1
        for d in range(len(other_dims)):
            M *= x_perm.shape[d]
        N = 1
        for d in range(len(other_dims), ndim):
            N *= x_perm.shape[d]

        x_2d = x_perm.reshape(M, N).contiguous()

        # --- Build and run kernel ---
        result_2d = _ext.run_sum(x_2d)  # (M, 1)

        # --- Reshape output to match expected shape ---
        leading_shape = [x.shape[d] for d in other_dims]

        if keepdim:
            out_shape = list(x.shape)
            for d in reduce_dims:
                out_shape[d] = 1
            result = result_2d.reshape(out_shape)
        else:
            result = result_2d.reshape(leading_shape)

        return result
