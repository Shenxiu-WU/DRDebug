import torch
import torch.nn as nn


SCENARIOS = [
    {
        "name": "last_dim_fp16_basic",
        "x_shape": (8, 32, 128),
        "index_shape": (8, 32, 64),
        "dim": 2,
        "dtype": torch.float16,
        "allow_negative_index": False,
        "expected_mode": "last_dim",
    },
    {
        "name": "last_dim_fp32_negative",
        "x_shape": (4, 97),
        "index_shape": (4, 33),
        "dim": -1,
        "dtype": torch.float32,
        "allow_negative_index": True,
        "expected_mode": "last_dim",
    },
    {
        "name": "middle_dim_fp16_post_large",
        "x_shape": (4, 128, 32),
        "index_shape": (4, 16, 32),
        "dim": 1,
        "dtype": torch.float16,
        "allow_negative_index": False,
        "expected_mode": "permute_last_dim",
    },
    {
        "name": "middle_dim_fp32_post_small",
        "x_shape": (4, 97, 2),
        "index_shape": (4, 15, 2),
        "dim": 1,
        "dtype": torch.float32,
        "allow_negative_index": False,
        "expected_mode": "permute_last_dim",
    },
    {
        "name": "dim0_fp16",
        "x_shape": (64, 8, 16),
        "index_shape": (12, 8, 16),
        "dim": 0,
        "dtype": torch.float16,
        "allow_negative_index": False,
        "expected_mode": "permute_last_dim",
    },
    {
        "name": "negative_dim_mid_fp32",
        "x_shape": (2, 5, 7, 16),
        "index_shape": (2, 3, 7, 16),
        "dim": -3,
        "dtype": torch.float32,
        "allow_negative_index": True,
        "expected_mode": "permute_last_dim",
    },
]


def normalize_dim(dim: int, rank: int) -> int:
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"Invalid dim={dim} for rank={rank}")
    return normalized


def build_scenario_key(x_shape, index_shape, dim: int, dtype) -> tuple:
    return (
        tuple(int(v) for v in x_shape),
        tuple(int(v) for v in index_shape),
        int(dim),
        str(dtype),
    )


SCENARIO_BY_KEY = {
    build_scenario_key(
        scenario["x_shape"],
        scenario["index_shape"],
        normalize_dim(scenario["dim"], len(scenario["x_shape"])),
        scenario["dtype"],
    ): scenario
    for scenario in SCENARIOS
}


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _resolve_scenario(self, x: torch.Tensor, index: torch.Tensor, dim: int):
        if x.ndim != index.ndim:
            raise ValueError(f"Expected x.ndim == index.ndim, got {x.ndim} and {index.ndim}")

        normalized_dim = normalize_dim(dim, x.ndim)
        key = build_scenario_key(x.shape, index.shape, normalized_dim, x.dtype)
        scenario = SCENARIO_BY_KEY.get(key)
        if scenario is None:
            supported = ", ".join(
                f"(x={case['x_shape']}, index={case['index_shape']}, dim={case['dim']}, dtype={case['dtype']})"
                for case in SCENARIOS
            )
            raise ValueError(
                f"Unsupported gather_elements_v2 case x={tuple(x.shape)}, index={tuple(index.shape)}, "
                f"dim={dim}, dtype={x.dtype}. Supported cases: {supported}"
            )
        return scenario, normalized_dim

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
        _, normalized_dim = self._resolve_scenario(x, index, dim)
        return torch.gather(x, normalized_dim, index.to(torch.int64))


def _make_x(shape, dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)


def _make_index(shape, gather_dim_size: int, allow_negative_index: bool, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if allow_negative_index:
        return torch.randint(
            -gather_dim_size,
            gather_dim_size,
            shape,
            generator=generator,
            dtype=torch.int32,
        )
    return torch.randint(
        0,
        gather_dim_size,
        shape,
        generator=generator,
        dtype=torch.int32,
    )


def get_input_groups():
    input_groups = []
    for index, scenario in enumerate(SCENARIOS):
        normalized_dim = normalize_dim(scenario["dim"], len(scenario["x_shape"]))
        gather_dim_size = scenario["x_shape"][normalized_dim]
        x = _make_x(scenario["x_shape"], scenario["dtype"], seed=2026 + index)
        gather_index = _make_index(
            scenario["index_shape"],
            gather_dim_size,
            scenario["allow_negative_index"],
            seed=3036 + index,
        )
        input_groups.append([x, gather_index, int(scenario["dim"])])
    return input_groups


def get_init_inputs():
    return []
