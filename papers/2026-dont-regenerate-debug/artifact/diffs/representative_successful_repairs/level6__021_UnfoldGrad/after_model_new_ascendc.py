import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _unfold_backward_ext as _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output, input_sizes, dim, size, step):
        dim = int(dim)
        size = int(size)
        step = int(step)
        D = int(input_sizes[dim])
        n_windows = (D - size) // step + 1
        if n_windows <= 0:
            return grad_output.new_zeros(input_sizes)

        grad_output_ndim = grad_output.ndim
        other_dims = [i for i in range(grad_output_ndim) if i != dim and i != grad_output_ndim - 1]
        perm = other_dims + [dim, grad_output_ndim - 1]
        grad_output_perm = grad_output.permute(perm).contiguous()

        B = 1
        for extent in grad_output_perm.shape[:-2]:
            B *= int(extent)
        grad_output_2d = grad_output_perm.reshape(B, grad_output_perm.shape[-2], grad_output_perm.shape[-1])

        if grad_output_2d.shape[1] > n_windows:
            grad_output_2d = grad_output_2d[:, :n_windows, :]

        grad_input_2d = _ext.run_unfold_backward(grad_output_2d, D, n_windows, size, step)

        out_prefix_shape = [int(v) for v in grad_output_perm.shape[:-2]]
        grad_input_perm = grad_input_2d.reshape(out_prefix_shape + [D])

        inverse_perm = [0] * grad_output_ndim
        for new_axis, old_axis in enumerate(perm):
            inverse_perm[old_axis] = new_axis
        grad_input = grad_input_perm.permute(inverse_perm[:-1]).contiguous()
        return grad_input
