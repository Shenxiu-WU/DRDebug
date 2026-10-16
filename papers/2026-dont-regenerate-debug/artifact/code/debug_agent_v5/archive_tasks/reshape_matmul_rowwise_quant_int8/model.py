import torch
import torch.nn as nn
import torch_npu


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError(f"h must be square, got shape={tuple(h.shape)}")
        k = h.shape[0]
        if n % k != 0:
            raise ValueError(f"x.shape[1] must be divisible by k={k}, got {n}")

        # Reshape view: split each input row into chunks of length k, then run GEMM.
        # This is equivalent to:
        #   (m, n) -> (m * (n / k), k) @ (k, k) -> (m * (n / k), k) -> (m, n)
        m_gemm = m * (n // k)
        result = torch.mm(x.reshape(m_gemm, k), h)
        result = result.reshape(m, n)

        # Row-wise dynamic int8 quantization implemented with torch ops:
        #   scale_i = max(abs(row_i)) / 127
        #   y_i = round(row_i / scale_i), clipped to int8 range
        y, scale = torch_npu.npu_dynamic_quant(result)
        return y


def get_input_groups():
    # Default sample keeps power-of-two shapes and ensures n > k.
    cases = [
        (4096, 2048, 1024),
    ]
    input_groups = []
    for m, n, k in cases:
        x = torch.rand(m, n, dtype=torch.bfloat16)
        h = torch.rand(k, k, dtype=torch.bfloat16)
        input_groups.append([x, h])
    return input_groups


def get_init_inputs():
    return []
