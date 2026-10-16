import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        acc = torch.einsum("bhsd,bhkd->bhsk", q, k) * (1.0 / q.shape[-1])**0.5
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v)
        return o.to(torch.float16)


def get_input_groups():
    cases = [
        (4, 4096, 32, 512),
    ]
    input_groups = []
    for b, s, h, d in cases:
        q = torch.rand((b, h, s, d), dtype=torch.float16)
        k = torch.rand((b, h, s, d), dtype=torch.float16)
        v = torch.rand((b, h, s, d), dtype=torch.float16)
        input_groups.append([q, k, v])
    return input_groups

def get_init_inputs():
    return []
