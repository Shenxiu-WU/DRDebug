"""AscendC implementation for MoE Init Routing.

The AscendC kernel materializes:
    expanded_x[r, :H] = x[expanded_row_idx[r], :H]

`expanded_row_idx` and `expanded_expert_idx` are produced by reshaping the
input tensors to 1D — these are pure layout views, no compute.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _moe_init_routing_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        row_idx: torch.Tensor,
        expert_idx: torch.Tensor,
        active_num: int,
    ):
        N = int(x.shape[0])
        H = int(x.shape[1])
        K = int(row_idx.shape[1])
        M = N * K

        # Layout-only operations: no compute is performed by these.
        x_contig = x.contiguous()
        expanded_row_idx = row_idx.contiguous().view(M).to(torch.int32)
        expanded_expert_idx = expert_idx.contiguous().view(M).to(torch.int32)

        # 32B-aligned inner stride for the kernel UB transfer.
        element_bytes = 2 if x.dtype in (torch.float16, torch.bfloat16) else 4
        per_row_bytes = ((H * element_bytes + 31) // 32) * 32
        XStride = per_row_bytes // element_bytes
        YStride = XStride

        # Build a contiguous, padded source. We only zero-fill the padding
        # columns; the H useful columns are populated by a slice assignment
        # which is a tensor-layout move, not a compute op.
        x_padded = x.new_zeros((N, XStride))
        x_padded[:, :H] = x_contig

        y_padded = _ext.run_moe_init_routing(
            x_padded, expanded_row_idx, H, XStride, YStride
        )

        expanded_x = y_padded[:, :H].contiguous()
        return expanded_x, expanded_row_idx, expanded_expert_idx
