import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert k.shape == v.shape
        assert q.shape[0] == k.shape[0]
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert q.shape[1] % k.shape[1] == 0

        group_size = q.shape[1] // k.shape[1]
        k_expanded = k.repeat_interleave(group_size, dim=1)
        v_expanded = v.repeat_interleave(group_size, dim=1)

        acc = torch.einsum("bhsd,bhkd->bhsk", q, k_expanded) * (1.0 / q.shape[-1]) ** 0.5
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v_expanded)
        return o.to(torch.float16)


def get_input_groups():
    cases = [
        (2, 2048, 4096, 32, 8, 128),
    ]
    input_groups = []
    for b, q_s, kv_s, h, kv_h, d in cases:
        q = torch.rand((b, h, q_s, d), dtype=torch.float16)
        k = torch.rand((b, kv_h, kv_s, d), dtype=torch.float16)
        v = torch.rand((b, kv_h, kv_s, d), dtype=torch.float16)
        input_groups.append([q, k, v])
    return input_groups


def get_init_inputs():
    return []
