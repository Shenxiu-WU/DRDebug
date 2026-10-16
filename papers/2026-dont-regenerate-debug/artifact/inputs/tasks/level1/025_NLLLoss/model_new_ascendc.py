import torch
import torch.nn as nn
import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _nllloss_ext as _ext


class Model(nn.Module):
    """
    NLLLoss implemented with custom AscendC kernel.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor = None, ignore_index: int = -100,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the negative log likelihood loss using AscendC kernel.
        """
        reduction_map = {"none": 0, "sum": 1, "mean": 2}
        reduction_val = reduction_map[reduction]

        if weight is None:
            weight = torch.empty(0, dtype=input.dtype, device=input.device)

        C = input.shape[1]
        input_flat = input.reshape(-1, C).contiguous()
        target_flat = target.reshape(-1).contiguous().to(torch.int32)

        orig_dtype = input.dtype
        if orig_dtype == torch.bfloat16:
            input_f = input_flat.to(torch.float32)
            weight_f = weight.to(torch.float32) if weight.numel() > 0 else weight
            out = _ext.run_nllloss(input_f, target_flat, weight_f, ignore_index, reduction_val)
            out = out.to(torch.bfloat16)
        else:
            out = _ext.run_nllloss(input_flat, target_flat, weight, ignore_index, reduction_val)

        if reduction == "none":
            return out.reshape(target.shape)
        return out
