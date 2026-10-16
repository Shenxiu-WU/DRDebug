#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "masked_softmax_attn_dropout_bwd_tiling.h"

extern "C" void masked_softmax_attn_dropout_bwd_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *pAttn, uint8_t *mask,
    uint8_t *dropoutMask, uint8_t *out, uint8_t *tiling);

extern "C" void masked_softmax_attn_dropout_bwd_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *pAttn, uint8_t *mask,
    uint8_t *dropoutMask, uint8_t *out, uint8_t *tiling);

extern "C" void masked_softmax_attn_dropout_bwd_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *pAttn, uint8_t *mask,
    uint8_t *dropoutMask, uint8_t *out, uint8_t *tiling);

namespace masked_softmax_attn_dropout_bwd_ext {

using LaunchFn = void (*)(
    uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::object run_masked_softmax_attn_dropout_bwd(
    const at::Tensor &grad_output,
    const at::Tensor &p_attn,
    const at::Tensor &mask,
    const at::Tensor &dropout_mask,
    double p_dropout)
{
    TORCH_CHECK(grad_output.dim() == 4, "grad_output must be [B, H, T, T]");
    TORCH_CHECK(p_attn.dim() == 4, "p_attn must be [B, H, T, T]");
    TORCH_CHECK(mask.dim() == 4, "mask must be [B, H, T, T]");
    TORCH_CHECK(dropout_mask.dim() == 4, "dropout_mask must be [B, H, T, T]");
    TORCH_CHECK(
        grad_output.scalar_type() == at::kFloat ||
        grad_output.scalar_type() == at::kHalf ||
        grad_output.scalar_type() == at::kBFloat16,
        "grad_output must be float16, float32, or bfloat16");
    TORCH_CHECK(p_attn.scalar_type() == grad_output.scalar_type(), "p_attn dtype mismatch");
    TORCH_CHECK(mask.scalar_type() == grad_output.scalar_type(), "mask dtype mismatch");
    TORCH_CHECK(dropout_mask.scalar_type() == grad_output.scalar_type(), "dropout_mask dtype mismatch");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(p_attn.is_contiguous(), "p_attn must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(dropout_mask.is_contiguous(), "dropout_mask must be contiguous");

    const int32_t B = static_cast<int32_t>(grad_output.sizes()[0]);
    const int32_t H = static_cast<int32_t>(grad_output.sizes()[1]);
    const int32_t T = static_cast<int32_t>(grad_output.sizes()[2]);
    const int32_t M = B * H * T;

    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, M);
    const int32_t tasksPerCore = (M + usedCoreNum - 1) / usedCoreNum;

    at::Tensor out = at::empty_like(grad_output);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(MaskedSoftmaxAttnDropoutBwdTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<MaskedSoftmaxAttnDropoutBwdTiling *>(tilingCpu.data_ptr());
    tiling->M = M;
    tiling->T = T;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->p_dropout = static_cast<float>(p_dropout);
    tiling->scale = (p_dropout > 0.0) ? static_cast<float>(1.0 / (1.0 - p_dropout)) : 1.0f;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (grad_output.scalar_type() == at::kFloat) {
        launch = masked_softmax_attn_dropout_bwd_do_fp32;
    } else if (grad_output.scalar_type() == at::kHalf) {
        launch = masked_softmax_attn_dropout_bwd_do_fp16;
    } else if (grad_output.scalar_type() == at::kBFloat16) {
        launch = masked_softmax_attn_dropout_bwd_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(grad_output.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(p_attn.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(mask.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(dropout_mask.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return pybind11::cast(out);
}

} // namespace masked_softmax_attn_dropout_bwd_ext

PYBIND11_MODULE(_masked_softmax_attn_dropout_bwd_ext, m)
{
    m.doc() = "masked_softmax_with_attention_dropout_backward extension";
    m.def("run_masked_softmax_attn_dropout_bwd",
          &masked_softmax_attn_dropout_bwd_ext::run_masked_softmax_attn_dropout_bwd, "");
}
