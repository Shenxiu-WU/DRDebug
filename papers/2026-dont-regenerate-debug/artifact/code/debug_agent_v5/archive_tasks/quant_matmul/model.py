import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor):
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        out = out * scale
        return out.to(torch.float16)


def get_input_groups():
    cases = [
        (1024, 1024, 1024),
    ]
    input_groups = []
    for m, n, k in cases:
        a = torch.randint(-128, 127, (m, k), dtype=torch.int8)
        b = torch.randint(-128, 127, (k, n), dtype=torch.int8)
        scale = torch.randn(n, dtype=torch.float32)
        input_groups.append([a, b, scale])
    return input_groups


def get_init_inputs():
    return []
