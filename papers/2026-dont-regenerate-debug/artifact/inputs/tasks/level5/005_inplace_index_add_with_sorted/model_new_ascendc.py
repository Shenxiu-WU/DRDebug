import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import torch
import torch.nn as nn

import _inplace_index_add_ext as _ext


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var, value, sorted_indices, pos, alpha, axis):
        # pos is unused in this implementation
        out = var.clone()

        # Handle non-zero axis by moving it to position 0
        if axis != 0:
            perm = [axis] + [i for i in range(var.ndim) if i != axis]
            out = out.permute(perm).contiguous()
            value = value.permute([axis] + [i for i in range(value.ndim) if i != axis]).contiguous()

        # Flatten to 2D: [M, N] and [K, N]
        M = int(out.shape[0])
        N = int(out.numel() // M)
        K = int(value.shape[0])

        out_2d = out.reshape(M, N)
        value_2d = value.reshape(K, N)

        # Prepare alpha as scalar tensor with same dtype as out
        alpha_val = alpha.item() if alpha.numel() == 1 else 1.0
        alpha_tensor = torch.tensor([alpha_val], dtype=out.dtype, device=out.device)

        out_2d = _ext.run_inplace_index_add(out_2d, value_2d, sorted_indices.to(torch.int32), alpha_tensor)

        # Reshape back to permuted shape
        out = out_2d.reshape(out.shape)

        if axis != 0:
            inverse_perm = [0] * var.ndim
            for new_axis, old_axis in enumerate(perm):
                inverse_perm[old_axis] = new_axis
            out = out.permute(inverse_perm).contiguous()

        return out
