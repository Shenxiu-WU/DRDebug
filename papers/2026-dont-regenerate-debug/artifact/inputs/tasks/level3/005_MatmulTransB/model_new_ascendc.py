"""AscendC-backed matmul with transposed B (C = A @ B^T)."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _matmul_transb_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    """AscendC-backed Matmul with transposed B."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2-D"
        assert A.shape[1] == B.shape[1], "K dimension must match"
        assert A.dtype == B.dtype, "A and B must have the same dtype"
        assert A.dtype in (torch.float32, torch.float16, torch.bfloat16), (
            f"unsupported dtype: {A.dtype}"
        )
        return _ext.run_matmul_transb(A.contiguous(), B.contiguous())


def get_init_inputs():
    return []
