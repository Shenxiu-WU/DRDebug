import sys
from pathlib import Path

import torch
import torch.nn as nn

from gather_elements_v2.model import normalize_dim

TRANSPOSE_MODE_MAX_X_GATHER = 2048
TRANSPOSE_MODE_MAX_IDX_GATHER = 2048

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _current_task_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _resolve_dim(self, x: torch.Tensor, index: torch.Tensor, dim: int) -> int:
        if x.ndim != index.ndim:
            raise ValueError(f"Expected x.ndim == index.ndim, got {x.ndim} and {index.ndim}")
        return normalize_dim(dim, x.ndim)

    def _choose_mode(self, x_prefix_shape, idx_prefix_shape, x_gather_dim: int, idx_gather_dim: int) -> int:
        # 0: last_dim, 1: transpose, 2: indexed(scalar-like fallback)
        if x_prefix_shape == idx_prefix_shape:
            return 0

        idx_rows = 1
        for extent in idx_prefix_shape:
            idx_rows *= int(extent)

        if idx_rows >= 64 and x_gather_dim <= TRANSPOSE_MODE_MAX_X_GATHER and idx_gather_dim <= TRANSPOSE_MODE_MAX_IDX_GATHER:
            return 1
        return 2

    def _compute_row_map(self, x_prefix_shape, idx_prefix_shape, device):
        if len(idx_prefix_shape) == 0:
            return torch.zeros((1,), dtype=torch.int32, device=device)

        idx_rows = 1
        for extent in idx_prefix_shape:
            idx_rows *= int(extent)

        x_prefix_strides = []
        running = 1
        for extent in reversed(x_prefix_shape):
            x_prefix_strides.append(running)
            running *= int(extent)
        x_prefix_strides = list(reversed(x_prefix_strides))

        idx_prefix_strides = []
        running = 1
        for extent in reversed(idx_prefix_shape):
            idx_prefix_strides.append(running)
            running *= int(extent)
        idx_prefix_strides = list(reversed(idx_prefix_strides))

        row_ids = torch.arange(idx_rows, device=device, dtype=torch.int64)
        row_map = torch.zeros((idx_rows,), device=device, dtype=torch.int64)
        tmp = row_ids
        for axis, idx_stride in enumerate(idx_prefix_strides):
            coord = tmp // int(idx_stride)
            tmp = tmp - coord * int(idx_stride)
            row_map = row_map + coord * int(x_prefix_strides[axis])

        return row_map.to(torch.int32)

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int):
        normalized_dim = self._resolve_dim(x, index, dim)

        assert x.dtype in (torch.float16, torch.float32)
        assert index.dtype in (torch.int32, torch.int64)
        for axis, (x_extent, index_extent) in enumerate(zip(x.shape, index.shape)):
            if axis != normalized_dim:
                assert index_extent <= x_extent

        perm = [axis for axis in range(x.ndim) if axis != normalized_dim] + [normalized_dim]
        inverse_perm = [0] * x.ndim
        for new_axis, old_axis in enumerate(perm):
            inverse_perm[old_axis] = new_axis

        x_perm = x.permute(perm).contiguous()
        index_perm = index.permute(perm).contiguous().to(torch.int32)
        index_perm = torch.where(index_perm < 0, index_perm + x_perm.shape[-1], index_perm)

        x_prefix_shape = tuple(int(v) for v in x_perm.shape[:-1])
        idx_prefix_shape = tuple(int(v) for v in index_perm.shape[:-1])

        x_rows = 1
        for extent in x_prefix_shape:
            x_rows *= extent
        idx_rows = 1
        for extent in idx_prefix_shape:
            idx_rows *= extent

        x_gather_dim = int(x_perm.shape[-1])
        idx_gather_dim = int(index_perm.shape[-1])

        if idx_rows == 0 or idx_gather_dim == 0:
            return x.new_empty(index.shape)

        element_bytes = 2 if x.dtype == torch.float16 else 4
        x_stride = ((x_gather_dim * element_bytes + 31) // 32) * (32 // element_bytes)
        y_stride = ((idx_gather_dim * 4 + 31) // 32) * 8

        x_rows_2d = x_perm.reshape(x_rows, x_gather_dim)
        index_rows_2d = index_perm.reshape(idx_rows, idx_gather_dim)

        x_padded = x.new_zeros((x_rows, x_stride))
        x_padded[:, :x_gather_dim] = x_rows_2d

        index_padded = index_rows_2d.new_zeros((idx_rows, y_stride))
        index_padded[:, :idx_gather_dim] = index_rows_2d

        mode = self._choose_mode(x_prefix_shape, idx_prefix_shape, x_gather_dim, idx_gather_dim)
        if mode == 0:
            row_map = torch.zeros((idx_rows,), dtype=torch.int32, device=index.device)
        else:
            row_map = self._compute_row_map(x_prefix_shape, idx_prefix_shape, index.device)

        y_padded = _ext.run_gather_elements_v2(x_padded, index_padded, row_map, idx_gather_dim, mode)
        y_flat = y_padded[:, :idx_gather_dim]
        y_perm = y_flat.reshape(index_perm.shape)
        return y_perm.permute(inverse_perm).contiguous()
