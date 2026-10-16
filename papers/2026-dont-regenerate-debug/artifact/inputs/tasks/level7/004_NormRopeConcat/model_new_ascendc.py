import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _norm_rope_concat_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoderQuery,
        encoderKey,
        encoderValue,
        normQueryWeight,
        normQueryBias,
        normKeyWeight,
        normKeyBias,
        normAddedQueryWeight,
        normAddedQueryBias,
        normAddedKeyWeight,
        normAddedKeyBias,
        ropeSin: torch.Tensor,
        ropeCos: torch.Tensor,
        normType: int,
        normAddedType: int,
        ropeType: int,
        concatOrder: int,
        eps: float,
        isTraining: bool,
    ):
        B, S, N, D = query.shape
        device = query.device
        dtype = query.dtype

        # Placeholder weight/bias if None
        zero_w = torch.ones(D, dtype=dtype, device=device)
        zero_b = torch.zeros(D, dtype=dtype, device=device)
        if normQueryWeight is None:
            normQueryWeight = zero_w
        if normQueryBias is None:
            normQueryBias = zero_b
        if normKeyWeight is None:
            normKeyWeight = zero_w
        if normKeyBias is None:
            normKeyBias = zero_b

        q_out, k_out, v_out = _ext.run_norm_rope_concat(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            normQueryWeight.contiguous(),
            normQueryBias.contiguous(),
            normKeyWeight.contiguous(),
            normKeyBias.contiguous(),
            ropeSin.contiguous(),
            ropeCos.contiguous(),
            int(normType),
            int(ropeType),
            float(eps),
        )
        return q_out, k_out, v_out

