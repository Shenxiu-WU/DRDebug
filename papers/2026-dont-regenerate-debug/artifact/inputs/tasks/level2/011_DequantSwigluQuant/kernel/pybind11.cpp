#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cstdint>
#include <tuple>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "dequant_swiglu_quant_tiling.h"

extern "C" void dequant_swiglu_quant_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight_scale, uint8_t *activation_scale,
    uint8_t *quant_scale, uint8_t *y, uint8_t *scales, uint8_t *tiling);

namespace my_dequant_swiglu_quant {

std::tuple<at::Tensor, at::Tensor> run_dequant_swiglu_quant(
    const at::Tensor &x,
    const at::Tensor &weight_scale,
    const at::Tensor &activation_scale,
    const at::Tensor &quant_scale,
    bool activate_left,
    int swiglu_mode,
    float clamp_limit,
    float glu_alpha,
    float glu_bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == at::kInt, "x must be int32");
    int32_t T = x.sizes()[0];
    int32_t twoH = x.sizes()[1];
    TORCH_CHECK(twoH % 2 == 0, "x last dim must be even");
    int32_t H = twoH / 2;

    at::Tensor ws = weight_scale;
    at::Tensor act = activation_scale;
    at::Tensor qs = quant_scale;

    if (ws.device().is_cpu()) {
        ws = ws.to(at::kPrivateUse1);
    }
    if (act.device().is_cpu()) {
        act = act.to(at::kPrivateUse1);
    }
    if (qs.device().is_cpu()) {
        qs = qs.to(at::kPrivateUse1);
    }

    TORCH_CHECK(ws.dim() == 2 && ws.sizes()[1] == twoH, "weight_scale shape mismatch");
    TORCH_CHECK(act.dim() == 2 && act.sizes()[0] == T && act.sizes()[1] == 1, "activation_scale shape mismatch");
    TORCH_CHECK(qs.dim() == 2 && qs.sizes()[1] == H, "quant_scale shape mismatch");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t blockDim = static_cast<uint32_t>(T < 24 ? T : 24);

    at::Tensor y = at::empty({T, H}, at::device(at::kPrivateUse1).dtype(at::kChar));
    at::Tensor scales = at::empty({T}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor t = at::empty({(int64_t)sizeof(DequantSwigluQuantTiling)},
                              at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<DequantSwigluQuantTiling *>(t.data_ptr());
    tp->T = T;
    tp->twoH = twoH;
    tp->H = H;
    tp->swiglu_mode = swiglu_mode;
    tp->activate_left = activate_left ? 1 : 0;
    tp->clamp_limit = clamp_limit;
    tp->glu_alpha = glu_alpha;
    tp->glu_bias = glu_bias;
    auto tiling_npu = t.to(at::kPrivateUse1);

    dequant_swiglu_quant_do(
        blockDim, acl_stream,
        static_cast<uint8_t*>(const_cast<void*>(x.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(ws.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(act.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(qs.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(y.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(scales.storage().data())),
        static_cast<uint8_t*>(const_cast<void*>(tiling_npu.storage().data())));

    return std::make_tuple(y, scales);
}

} // namespace my_dequant_swiglu_quant

PYBIND11_MODULE(_dequant_swiglu_quant_ext, m)
{
    m.doc() = "dequant_swiglu_quant pybind11 interfaces";
    m.def("run_dequant_swiglu_quant", &my_dequant_swiglu_quant::run_dequant_swiglu_quant, "");
}
