import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

# Import compiled AscendC extension from kernel/build
import importlib.util

_BUILD_DIR = _TASK_DIR / "kernel" / "build"
_ext_name = "_tanh_gated_residual_add_backward_ext"
_ext_so = None
for f in _BUILD_DIR.glob(f"{_ext_name}*.so"):
    _ext_so = f
    break

if _ext_so is None:
    raise ImportError(f"Cannot find compiled AscendC extension {_ext_name} in {_BUILD_DIR}")

_spec = importlib.util.spec_from_file_location(_ext_name, _ext_so)
_ext = importlib.util.module_from_spec(_spec)
_sys_mod = sys.modules.get(_ext_name)
if _sys_mod is not None:
    _ext = _sys_mod
else:
    sys.modules[_ext_name] = _ext
    _spec.loader.exec_module(_ext)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        gate: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ):
        assert grad_output.shape == gate.shape == hidden_states.shape == mask.shape
        assert grad_output.dtype == gate.dtype == hidden_states.dtype == mask.dtype
        assert grad_output.dtype == torch.bfloat16

        original_shape = grad_output.shape

        # Flatten to 1D (contiguous)
        go_1d = grad_output.reshape(-1).contiguous()
        gate_1d = gate.reshape(-1).contiguous()
        hs_1d = hidden_states.reshape(-1).contiguous()
        mask_1d = mask.reshape(-1).contiguous()

        # Launch AscendC kernel
        grad_residual_1d, grad_hidden_states_1d, grad_gate_1d = _ext.run_tanh_gated_residual_add_backward(
            go_1d, gate_1d, hs_1d, mask_1d
        )

        # Reshape back
        grad_residual = grad_residual_1d.reshape(original_shape)
        grad_hidden_states = grad_hidden_states_1d.reshape(original_shape)
        grad_gate = grad_gate_1d.reshape(original_shape)

        return grad_residual, grad_hidden_states, grad_gate
