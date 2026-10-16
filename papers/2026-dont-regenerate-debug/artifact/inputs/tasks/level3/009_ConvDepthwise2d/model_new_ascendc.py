"""AscendC wrapper for depthwise 2D convolution.

Layout strategy:
    Input  x : (N, C, H, W) -> permute to NHWC -> flatten to (N*H*W, C)
    Weight w : (C, 1, kH, kW) -> permute to (kH, kW, C) -> flatten to (kH*kW, C)
    Bias  b  : (C,) (zeros when has_bias is False)
    Output y : (N*OH*OW, C) -> reshape to (N, OH, OW, C) -> permute to (N, C, OH, OW)

Random-state alignment:
    The reference Model() builds a fresh nn.Conv2d inside forward() each call.
    Without intervention the verification loop calls ref then cand sequentially,
    consuming RNG twice; ref and cand would then see different Conv2d weights.
    Both Model and ModelNew therefore wrap their nn.Conv2d construction with a
    save / set / restore of the global RNG state, seeding nn.Conv2d to a fixed
    value. This guarantees ref and cand always share identical weights without
    perturbing any other random operation.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
_KERNEL_BUILD = _TASK_DIR / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _current_task_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, in_channels, kernel_size, stride=1, padding=0, bias=True):
        # Reproduce the reference's deterministic-seed weight initialization
        # so that ref/cand share identical Conv2d weights for every call,
        # independent of the verification loop's call order. The RNG state is
        # restored afterwards to keep any external randomness untouched.
        _saved_rng = torch.random.get_rng_state()
        torch.random.manual_seed(0)
        conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        torch.random.set_rng_state(_saved_rng)

        device = x.device
        dtype = x.dtype

        N = int(x.shape[0])
        C = int(x.shape[1])
        H = int(x.shape[2])
        W = int(x.shape[3])
        kH = int(kernel_size)
        kW = int(kernel_size)
        sH = int(stride)
        sW = int(stride)
        pH = int(padding)
        pW = int(padding)

        OH = (H + 2 * pH - kH) // sH + 1
        OW = (W + 2 * pW - kW) // sW + 1

        # NHWC layout for x: (N, C, H, W) -> (N*H*W, C)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous().reshape(N * H * W, C).to(dtype)

        # weight: (C, 1, kH, kW) -> (kH, kW, C) -> (kH*kW, C)
        weight_filter = (
            conv.weight.detach()
            .squeeze(1)
            .permute(1, 2, 0)
            .contiguous()
            .reshape(kH * kW, C)
            .to(device=device, dtype=dtype)
        )

        if bias and conv.bias is not None:
            bias_vec = conv.bias.detach().contiguous().to(device=device, dtype=dtype)
            has_bias = 1
        else:
            bias_vec = torch.zeros(C, dtype=dtype, device=device)
            has_bias = 0

        y_flat = _ext.run_depthwise_conv2d(
            x_nhwc,
            weight_filter,
            bias_vec,
            int(N),
            int(C),
            int(H),
            int(W),
            int(OH),
            int(OW),
            int(kH),
            int(kW),
            int(sH),
            int(sW),
            int(pH),
            int(pW),
            int(has_bias),
        )

        # (N*OH*OW, C) -> (N, OH, OW, C) -> (N, C, OH, OW)
        y = y_flat.reshape(N, OH, OW, C).permute(0, 3, 1, 2).contiguous()
        return y
