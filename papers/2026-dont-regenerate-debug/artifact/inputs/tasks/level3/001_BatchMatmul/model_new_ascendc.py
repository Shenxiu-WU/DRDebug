import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _batch_matmul_ext as _ext


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.ndim == 3 and B.ndim == 3
        assert A.shape[0] == B.shape[0]
        assert A.shape[2] == B.shape[1]
        assert A.dtype == B.dtype

        orig_dtype = A.dtype
        if A.dtype != torch.float16:
            A_input = A.half()
            B_input = B.half()
        else:
            A_input = A
            B_input = B

        B_batch, M, K = A_input.shape
        _, _, N = B_input.shape

        block_M = 16
        block_N = 16
        block_K = 16

        pad_M = (block_M - M % block_M) % block_M
        pad_N = (block_N - N % block_N) % block_N
        pad_K = (block_K - K % block_K) % block_K

        if pad_M > 0 or pad_N > 0 or pad_K > 0:
            A_padded = torch.zeros((B_batch, M + pad_M, K + pad_K), dtype=A_input.dtype, device=A_input.device)
            A_padded[:, :M, :K] = A_input
            B_padded = torch.zeros((B_batch, K + pad_K, N + pad_N), dtype=B_input.dtype, device=B_input.device)
            B_padded[:, :K, :N] = B_input
            C_padded = _ext.run_batch_matmul(A_padded, B_padded)
            return C_padded.to(orig_dtype)[:, :M, :N]
        else:
            C = _ext.run_batch_matmul(A_input, B_input)
            return C.to(orig_dtype)
