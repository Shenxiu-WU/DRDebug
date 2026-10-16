import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import torch
import torch.nn as nn

import _moe_finalize_routing_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, expanded_permuted_rows, skip1, skip2, bias, scales,
                expanded_src_to_dst_row, export_for_source_row, drop_pad_mode=0):
        dtype = expanded_permuted_rows.dtype
        device = expanded_permuted_rows.device
        is_bf16 = (dtype == torch.bfloat16)

        if is_bf16:
            expanded_permuted_rows = expanded_permuted_rows.to(torch.float32)
            scales = scales.to(torch.float32)
            if bias is not None and bias.numel() > 0:
                bias = bias.to(torch.float32)
            if skip1 is not None and skip1.numel() > 0:
                skip1 = skip1.to(torch.float32)
            if skip2 is not None and skip2.numel() > 0:
                skip2 = skip2.to(torch.float32)
            work_dtype = torch.float32
        else:
            work_dtype = dtype

        if bias is None or bias.numel() == 0:
            bias = torch.empty((0,), dtype=work_dtype, device=device)
        if skip1 is None or skip1.numel() == 0:
            skip1 = torch.empty((0,), dtype=work_dtype, device=device)
        if skip2 is None or skip2.numel() == 0:
            skip2 = torch.empty((0,), dtype=work_dtype, device=device)

        out = _ext.run_moe_finalize_routing(
            expanded_permuted_rows, skip1, skip2, bias, scales,
            expanded_src_to_dst_row, export_for_source_row, drop_pad_mode
        )

        if is_bf16:
            out = out.to(torch.bfloat16)
        return out
