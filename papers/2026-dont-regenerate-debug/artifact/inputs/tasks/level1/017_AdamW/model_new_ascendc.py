"""AscendC implementation of AdamW (apply_adam) operator wrapper.

This module imports the compiled AscendC kernel extension `_adamw_ext` and
invokes it from `ModelNew.forward`. No element-wise computation runs in
PyTorch — every arithmetic step happens inside the AscendC kernel.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the compiled AscendC extension importable.
_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _adamw_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        grad: torch.Tensor,
        beta1_power,
        beta2_power,
        lr,
        beta1,
        beta2,
        epsilon,
        use_locking: bool = False,
        use_nesterov: bool = False,
    ):
        assert var.shape == grad.shape == m.shape == v.shape, "shape mismatch"
        assert var.dtype == grad.dtype, "var/grad dtype mismatch"
        assert var.dtype == m.dtype, "var/m dtype mismatch"
        assert var.dtype == v.dtype, "var/v dtype mismatch"

        original_shape = var.shape
        var_flat = var.contiguous().view(-1)
        m_flat = m.contiguous().view(-1)
        v_flat = v.contiguous().view(-1)
        grad_flat = grad.contiguous().view(-1)

        var_out, m_out, v_out = _ext.run_adamw(
            var_flat,
            m_flat,
            v_flat,
            grad_flat,
            float(beta1_power),
            float(beta2_power),
            float(lr),
            float(beta1),
            float(beta2),
            float(epsilon),
            bool(use_nesterov),
        )
        return (
            var_out.view(original_shape),
            m_out.view(original_shape),
            v_out.view(original_shape),
        )
