import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _time_decay_exp_stabilization_ext as _ext

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.attention_hidden_size = 2048

    def forward(
        self,
        time_decay: torch.Tensor,
        key: torch.Tensor,
        time_first: torch.Tensor,
        value: torch.Tensor,
        max_state: torch.Tensor,
        num_state: torch.Tensor,
        den_state: torch.Tensor,
    ):
        # Ensure contiguous and float32 for kernel
        time_decay_f = time_decay.contiguous().to(torch.float32)
        key_f = key.contiguous().to(torch.float32)
        time_first_f = time_first.contiguous().to(torch.float32)
        value_f = value.contiguous().to(torch.float32)
        max_state_f = max_state.contiguous().to(torch.float32)
        num_state_f = num_state.contiguous().to(torch.float32)
        den_state_f = den_state.contiguous().to(torch.float32)

        output, max_state_out, num_state_out, den_state_out = _ext.run_time_decay_exp_stabilization(
            time_decay_f, key_f, time_first_f, value_f, max_state_f, num_state_f, den_state_f
        )

        # Cast back to original dtype if needed
        if time_decay.dtype == torch.float16:
            output = output.to(torch.float16)
            max_state_out = max_state_out.to(torch.float16)
            num_state_out = num_state_out.to(torch.float16)
            den_state_out = den_state_out.to(torch.float16)
        elif time_decay.dtype == torch.bfloat16:
            output = output.to(torch.bfloat16)
            max_state_out = max_state_out.to(torch.bfloat16)
            num_state_out = num_state_out.to(torch.bfloat16)
            den_state_out = den_state_out.to(torch.bfloat16)

        return output, max_state_out, num_state_out, den_state_out
