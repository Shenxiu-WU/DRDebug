import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_build_dir = os.path.join(_current_dir, "kernel", "build")
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)

import torch
import torch.nn as nn
import _hyena_rfft_ext

class ModelNew(nn.Module):
    """
    AscendC-optimized Hyena FFT size padding + RFFT + normalize.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, seqlen].
        Returns:
            tuple: (x_freq_real, x_freq_imag) each with shape [batch_size, channels, seqlen+1].
        """
        x_f32 = x.float()
        real, imag = _hyena_rfft_ext.run_hyena_rfft(x_f32)
        return real, imag
