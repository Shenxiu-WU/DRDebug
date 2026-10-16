"""AscendC-backed MatmulTransA implementation.

The Python wrapper mirrors the TileLang version: it materialises
`A_T = A.T.contiguous()` (a pure layout transform — allowed by the
no-torch-compute rule), zero-pads `A_T` and `B` so all shape extents
become multiples of 16, calls the AscendC cube kernel, and then slices
the padded output back to the original `(n, k)` shape.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
_KERNEL_BUILD = _TASK_DIR / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _matmul_trans_a_ext as _ext  # noqa: E402


_BASE = 16  # cube fractal / baseM=baseN=baseK alignment
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _pad_to(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
        assert A.shape[0] == B.shape[0], "leading reduction dim must match"
        assert A.dtype == B.dtype, "A and B must share dtype"
        assert A.dtype in _SUPPORTED_DTYPES, (
            f"unsupported dtype {A.dtype}; expected float16/bfloat16/float32"
        )

        m, n = A.shape
        _, k = B.shape

        # Layout-only transform: A.T -> contiguous (n, m).
        A_T = A.t().contiguous()

        # Pad to multiples of `_BASE` so the cube kernel sees aligned shapes.
        M_pad = _pad_to(n, _BASE)
        K_pad = _pad_to(m, _BASE)
        N_pad = _pad_to(k, _BASE)

        if (M_pad, K_pad, N_pad) == (n, m, k):
            A_T_padded = A_T
            B_padded = B
        else:
            A_T_padded = torch.zeros(
                (M_pad, K_pad), dtype=A.dtype, device=A.device,
            )
            A_T_padded[:n, :m] = A_T

            B_padded = torch.zeros(
                (K_pad, N_pad), dtype=B.dtype, device=B.device,
            )
            B_padded[:m, :k] = B

        C_padded = _ext.run_matmul_trans_a(A_T_padded, B_padded)

        if (M_pad, N_pad) == (n, k):
            return C_padded
        return C_padded[:n, :k].contiguous()
