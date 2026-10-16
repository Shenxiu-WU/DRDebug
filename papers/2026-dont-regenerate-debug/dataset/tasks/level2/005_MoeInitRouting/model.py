import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    """
    Simple model that initializes routing for MoE (Mixture of Experts).
    torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num) -> (Tensor, Tensor, Tensor)
    Pure PyTorch implementation (replacing torch_npu.npu_moe_init_routing):

        def forward(self, x, row_idx, expert_idx, active_num):
            # x: [NUM_ROWS, H] input features
            # row_idx: [NUM_ROWS, K] original row positions
            # expert_idx: [NUM_ROWS, K] expert indices for each top-k
            # active_num: maximum number of rows to process

            NUM_ROWS, H = x.shape
            K = expert_idx.shape[1]

            # Flatten row_idx to get expanded row indices
            # expanded_row_idx: [NUM_ROWS * K]
            expanded_row_idx = row_idx.view(-1)

            # Flatten expert_idx to get expanded expert indices
            # expanded_expert_idx: [NUM_ROWS * K]
            expanded_expert_idx = expert_idx.view(-1)

            # Expand x by gathering rows according to expanded_row_idx
            # expanded_x: [NUM_ROWS * K, H]
            expanded_x = x[expanded_row_idx.long()]

            return expanded_x, expanded_row_idx, expanded_expert_idx
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, row_idx: torch.Tensor, expert_idx: torch.Tensor, active_num: int) -> tuple:
        """
        Initializes routing for MoE.

        Args:
            x (torch.Tensor): MOE input token features, must be 2D with shape (NUM_ROWS, H).
                              dtype: float16, bfloat16, float32, format: ND. Shape must be < 2^24.
            row_idx (torch.Tensor): Indicates the original row position for each position.
                                    Must have same shape as expert_idx. dtype: int32, format: ND.
            expert_idx (torch.Tensor): Output from npu_moe_gating_top_k_softmax indicating K experts 
                                       for each row feature. Must be 2D with shape (NUM_ROWS, K).
                                       dtype: int32, format: ND.
            active_num (int): Maximum number of rows to process.

        Returns:
            tuple: (expanded_x, expanded_row_idx, expanded_expert_idx) tensors for MoE routing.
        """
        return torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_MoeInitRouting.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        row_idx_info = inputs[1]
        expert_idx_info = inputs[2]
        active_num_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        x = torch.randn(x_info["shape"], dtype=dtype)
        row_idx = torch.randint(0, x_info["shape"][0], row_idx_info["shape"], dtype=torch.int32)
        expert_idx = torch.randint(0, 8, expert_idx_info["shape"], dtype=torch.int32)
        active_num = active_num_info["value"]
        input_groups.append([x, row_idx, expert_idx, active_num])
    return input_groups


def get_init_inputs():
    return []
