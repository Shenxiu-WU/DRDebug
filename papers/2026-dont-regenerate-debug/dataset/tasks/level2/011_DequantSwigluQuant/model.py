import json
import os
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs dequantization followed by SwiGLU and quantization.
    torch_npu.npu_dequant_swiglu_quant(x, *, weight_scale=None, activation_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0) -> (Tensor, Tensor)
    PyTorch native implementation of forward function
    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor = None, activation_scale: torch.Tensor = None,
                bias: torch.Tensor = None, quant_scale: torch.Tensor = None, quant_offset: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                swiglu_mode: int = 0, clamp_limit: float = 7.0, glu_alpha: float = 1.702,
                glu_bias: float = 1.0) -> tuple:
        # Dequantization
        x_float = x.float()

        if weight_scale is not None:
            x_float = x_float * weight_scale.float()

        if activation_scale is not None:
            x_float = x_float * activation_scale.float()

        if bias is not None:
            x_float = x_float + bias.float()

        # Split into two halves for SwiGLU
        out = torch.chunk(x_float, 2, dim=-1)

        if activate_left:
            self_tensor = out[0]
            other = out[1]
        else:
            self_tensor = out[1]
            other = out[0]

        # SwiGLU activation: F.silu(x) = x * sigmoid(x)
        output = F.silu(self_tensor) * other

        # Apply quant_scale: MULTIPLY
        if quant_scale is not None:
            output = output * quant_scale.float()

        # Quantization
        scale_dim0 = 1
        for s in x.shape[:-1]:
            scale_dim0 *= s

        if quant_mode == 0:  # Static
            if quant_offset is not None:
                output = output + quant_offset.float()
            output = torch.clamp(output, -128, 127)
            quantized_output = torch.round(output).to(torch.int8)
            quant_scales = torch.zeros(scale_dim0, dtype=torch.float32)

        elif quant_mode == 1:  # Dynamic
            abs_val = torch.abs(output)
            max_values = torch.amax(abs_val, dim=-1)
            quant_scales = max_values / 127.0
            quant_scales_clamped = torch.clamp(quant_scales, min=1e-10)
            output = output / quant_scales_clamped.unsqueeze(-1)
            output = torch.clamp(output, -128, 127)
            quantized_output = torch.round(output).to(torch.int8)
            quant_scales = quant_scales.reshape(scale_dim0)
        else:
            output = torch.clamp(output, -128, 127)
            quantized_output = torch.round(output).to(torch.int8)
            quant_scales = torch.zeros(scale_dim0, dtype=torch.float32)

        return quantized_output, quant_scales
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor = None, activation_scale: torch.Tensor = None,
                bias: torch.Tensor = None, quant_scale: torch.Tensor = None, quant_offset: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False, quant_mode: int = 0,
                swiglu_mode: int = 0, clamp_limit: float = 7.0, glu_alpha: float = 1.702,
                glu_bias: float = 1.0) -> tuple:
        """
        Performs dequantization followed by SwiGLU and quantization.

        Args:
            x (torch.Tensor): Target tensor. Must be 2D with shape [TokensNum, 2H], last axis even.
                              dtype: int32, bfloat16, format: ND.
            weight_scale (torch.Tensor, optional): Weight dequantization scale. Must be 2D [groupNum, 2H].
                                                   dtype: float32, format: ND. Required when x is int32.
            activation_scale (torch.Tensor, optional): Per-token weight dequantization scale.
                                                       Must be 1D [TokensNum]. dtype: float32.
                                                       Required when x is int32.
            bias (torch.Tensor, optional): Bias variable. dtype: int32, format: ND.
                                           Not effective when group_index is not None.
            quant_scale (torch.Tensor, optional): Smooth quantization scale. Must be 2D [groupNum, H].
                                                  dtype: float32, float16, bfloat16, format: ND.
            quant_offset (torch.Tensor, optional): Quantization offset.
                                                   dtype: float32, float16, bfloat16, format: ND.
                                                   Not effective when group_index is not None.
            group_index (Tensor, optional): Group tokens count (count mode only). Must be 1D.
                                            dtype: int64, format: ND.
            activate_left (bool, optional): Whether to activate left in SwiGLU. Default: False.
            quant_mode (int, optional): Quantization type. 0: static, 1: dynamic. Default: 0.
                                        When group_index is not None, only dynamic (1) is supported.
            swiglu_mode (int, optional): SwiGLU mode. 0: traditional, 1: variant with clamp/alpha/bias.
                                         Default: 0.
            clamp_limit (float, optional): SwiGLU output gate limit. Default: 7.0.
            glu_alpha (float, optional): GLU activation coefficient. Default: 1.702.
            glu_bias (float, optional): SwiGLU computation bias. Default: 1.0.

        Returns:
            tuple: (output tensor, quantization parameters) after dequant-SwiGLU-quant.
        """
        return torch_npu.npu_dequant_swiglu_quant(x, weight_scale=weight_scale, activation_scale=activation_scale,
                                                   bias=bias, quant_scale=quant_scale, quant_offset=quant_offset,
                                                   group_index=group_index, activate_left=activate_left,
                                                   quant_mode=quant_mode, swiglu_mode=swiglu_mode,
                                                   clamp_limit=clamp_limit, glu_alpha=glu_alpha, glu_bias=glu_bias)


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            attrs = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']

            x_shape = None
            x_dtype = None
            for inp in inputs:
                if inp['name'] == 'x' and inp['type'] == 'tensor':
                    x_shape = inp.get('shape')
                    x_dtype = inp.get('dtype', 'int32')
                    break

            if x_shape is not None:
                tokens_num = x_shape[0]
                hidden_size = x_shape[1]
                half_h = hidden_size // 2

                quant_mode = attrs.get('quant_mode', 0)
                group_index_val = tensors.get('group_index')
                has_group = group_index_val is not None

                if has_group:
                    num_groups = group_index_val.shape[0]
                    if quant_mode != 1:
                        quant_mode = 1
                    tensors['bias'] = None
                    tensors['quant_offset'] = None
                    tensors['group_index'] = torch.tensor([tokens_num // num_groups] * num_groups, dtype=torch.int64)
                else:
                    if x_dtype == 'int32':
                        if tensors.get('weight_scale') is not None:
                            ws_shape = tensors['weight_scale'].shape
                            tensors['weight_scale'] = torch.randn(ws_shape, dtype=torch.float32)
                        else:
                            tensors['weight_scale'] = torch.randn((1, hidden_size), dtype=torch.float32)
                        if tensors.get('activation_scale') is not None:
                            tensors['activation_scale'] = torch.randn((tokens_num, 1), dtype=torch.float32)
                        else:
                            tensors['activation_scale'] = torch.randn((tokens_num, 1), dtype=torch.float32)
                    else:
                        tensors['weight_scale'] = None
                        tensors['activation_scale'] = None
                        tensors['bias'] = None

                if tensors.get('quant_scale') is not None:
                    qs_shape = tensors['quant_scale'].shape
                    num_groups_qs = qs_shape[0] if len(qs_shape) > 0 else 1
                    tensors['quant_scale'] = torch.randn(num_groups_qs, half_h, dtype=torch.float32)

                if has_group:
                    tensors['quant_offset'] = None
                elif tensors.get('quant_offset') is not None and quant_mode == 0:
                    qs_tensor = tensors.get('quant_scale')
                    if qs_tensor is not None:
                        tensors['quant_offset'] = torch.randn(qs_tensor.shape, dtype=torch.float32)
                    else:
                        tensors['quant_offset'] = None
                else:
                    tensors['quant_offset'] = None

            group = [
                tensors['x'],
                tensors.get('weight_scale'),
                tensors.get('activation_scale'),
                tensors.get('bias'),
                tensors.get('quant_scale'),
                tensors.get('quant_offset'),
                tensors.get('group_index'),
                attrs.get('activate_left', False),
                quant_mode,
                attrs.get('swiglu_mode', 0),
                attrs.get('clamp_limit', 7.0),
                attrs.get('glu_alpha', 1.702),
                attrs.get('glu_bias', 1.0),
            ]
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
