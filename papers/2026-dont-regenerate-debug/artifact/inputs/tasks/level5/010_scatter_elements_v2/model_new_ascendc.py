import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _current_task_ext as _ext  # noqa: E402


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim {dim} out of range for tensor with {ndim} dimensions")
    return dim


def _resolve_reduction_mode(reduction: str) -> int:
    if reduction == "none":
        return 0
    elif reduction in ("add", "sum"):
        return 1
    elif reduction == "multiply":
        return 2
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, self_tensor: torch.Tensor, index: torch.Tensor, updates: torch.Tensor,
                axis: int = 0, reduction: str = "none") -> torch.Tensor:
        normalized_axis = _normalize_dim(axis, self_tensor.ndim)

        assert index.dtype in (torch.int32, torch.int64)
        assert updates.dtype == self_tensor.dtype
        for ax, (self_extent, idx_extent, upd_extent) in enumerate(zip(self_tensor.shape, index.shape, updates.shape)):
            if ax != normalized_axis:
                assert idx_extent <= self_extent
                assert upd_extent == idx_extent

        # Handle bfloat16 by converting to float32 for the kernel
        orig_dtype = self_tensor.dtype
        if orig_dtype == torch.bfloat16:
            self_tensor = self_tensor.to(torch.float32)
            updates = updates.to(torch.float32)

        # Move axis to last dimension
        perm = [ax for ax in range(self_tensor.ndim) if ax != normalized_axis] + [normalized_axis]
        inverse_perm = [0] * self_tensor.ndim
        for new_ax, old_ax in enumerate(perm):
            inverse_perm[old_ax] = new_ax

        self_perm = self_tensor.permute(perm).contiguous()
        index_perm = index.permute(perm).contiguous().to(torch.int32)
        updates_perm = updates.permute(perm).contiguous()

        # Flatten prefix dimensions
        prefix_shape = tuple(int(v) for v in self_perm.shape[:-1])
        idx_prefix_shape = tuple(int(v) for v in index_perm.shape[:-1])

        rows = 1
        for extent in prefix_shape:
            rows *= extent

        idx_rows = 1
        for extent in idx_prefix_shape:
            idx_rows *= extent

        x_gather_dim = int(self_perm.shape[-1])
        idx_gather_dim = int(index_perm.shape[-1])

        if rows == 0 or x_gather_dim == 0:
            out = self_tensor.clone()
            if orig_dtype == torch.bfloat16:
                out = out.to(torch.bfloat16)
            return out

        # Reshape to 2D
        self_rows_2d = self_perm.reshape(rows, x_gather_dim)
        index_rows_2d = index_perm.reshape(idx_rows, idx_gather_dim)
        updates_rows_2d = updates_perm.reshape(idx_rows, idx_gather_dim)

        element_bytes = 2 if self_tensor.dtype in (torch.float16, torch.bfloat16) else 4
        x_stride = ((x_gather_dim * element_bytes + 31) // 32) * (32 // element_bytes)
        y_stride = ((idx_gather_dim * element_bytes + 31) // 32) * (32 // element_bytes)

        # Pad to aligned stride
        self_padded = self_tensor.new_zeros((rows, x_stride))
        self_padded[:, :x_gather_dim] = self_rows_2d

        index_padded = index_rows_2d.new_zeros((idx_rows, y_stride))
        index_padded[:, :idx_gather_dim] = index_rows_2d

        updates_padded = updates_rows_2d.new_zeros((idx_rows, y_stride))
        updates_padded[:, :idx_gather_dim] = updates_rows_2d

        reduction_mode = _resolve_reduction_mode(reduction)
        out_padded = _ext.run_scatter_elements_v2(self_padded, index_padded, updates_padded, reduction_mode)
        out_flat = out_padded[:, :x_gather_dim]
        out_perm = out_flat.reshape(self_perm.shape)
        out = out_perm.permute(inverse_perm).contiguous()

        if orig_dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)
        return out
