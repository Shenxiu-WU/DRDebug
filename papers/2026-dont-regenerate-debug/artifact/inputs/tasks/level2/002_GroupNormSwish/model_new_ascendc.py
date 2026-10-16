import torch
import torch.nn as nn
import os
import sys

# Locate compiled extension
_kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel", "build")
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

import _group_norm_swish_ext


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor,
                eps: float = 1e-5, swish_scale: float = 1.0) -> tuple:
        output, mean, rstd = _group_norm_swish_ext.run_group_norm_swish(
            input, num_groups, weight, bias, eps, swish_scale)
        return output, mean, rstd
