import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs cumulative sum along a specified dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Applies cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int): The dimension to do the operation over.

        Returns:
            torch.Tensor: Output tensor with cumulative sum, same shape as input.
        """
        return torch.cumsum(x, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_Cumsum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        dim = dim_info["value"]
        input_groups.append([x, dim])
    return input_groups


def get_init_inputs():
    return []
