#include <algorithm>
#include <cstring>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "adamw_tiling.h"

extern "C" void adamw_apply_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *m,
    uint8_t *v,
    uint8_t *grad,
    uint8_t *varOut,
    uint8_t *mOut,
    uint8_t *vOut,
    uint8_t *tiling);

extern "C" void adamw_apply_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *m,
    uint8_t *v,
    uint8_t *grad,
    uint8_t *varOut,
    uint8_t *mOut,
    uint8_t *vOut,
    uint8_t *tiling);

extern "C" void adamw_apply_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *m,
    uint8_t *v,
    uint8_t *grad,
    uint8_t *varOut,
    uint8_t *mOut,
    uint8_t *vOut,
    uint8_t *tiling);

namespace adamw_ext {

using LaunchFn = void (*)(
    uint32_t,
    void *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *,
    uint8_t *);

pybind11::tuple run_adamw(
    const at::Tensor &var,
    const at::Tensor &m,
    const at::Tensor &v,
    const at::Tensor &grad,
    double beta1_power,
    double beta2_power,
    double lr,
    double beta1,
    double beta2,
    double epsilon,
    bool use_nesterov)
{
    TORCH_CHECK(var.is_contiguous(), "var must be contiguous");
    TORCH_CHECK(m.is_contiguous(), "m must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(var.numel() == grad.numel(), "var/grad numel mismatch");
    TORCH_CHECK(var.numel() == m.numel(), "var/m numel mismatch");
    TORCH_CHECK(var.numel() == v.numel(), "var/v numel mismatch");
    TORCH_CHECK(var.scalar_type() == grad.scalar_type(), "var/grad dtype mismatch");
    TORCH_CHECK(var.scalar_type() == m.scalar_type(), "var/m dtype mismatch");
    TORCH_CHECK(var.scalar_type() == v.scalar_type(), "var/v dtype mismatch");
    TORCH_CHECK(
        var.scalar_type() == at::kFloat
            || var.scalar_type() == at::kHalf
            || var.scalar_type() == at::kBFloat16,
        "var must be float32, float16, or bfloat16");

    const auto nTotal = static_cast<int32_t>(var.numel());
    constexpr int32_t blockN = static_cast<int32_t>(DEFAULT_BLOCK_N);
    constexpr int32_t subBlockN = static_cast<int32_t>(DEFAULT_SUB_BLOCK_N);
    const int32_t nNum = (nTotal + blockN - 1) / blockN;
    const int32_t usedCoreNum = std::min<int32_t>(
        static_cast<int32_t>(DEFAULT_NUM_PHYSICAL_CORES), std::max(1, nNum));
    const int32_t tasksPerCore = (nNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor varOut = at::empty_like(var);
    at::Tensor mOut = at::empty_like(m);
    at::Tensor vOut = at::empty_like(v);

    AdamWKernelTiling tiling{};
    tiling.nTotal = nTotal;
    tiling.blockN = blockN;
    tiling.subBlockN = subBlockN;
    tiling.usedCoreNum = usedCoreNum;
    tiling.tasksPerCore = tasksPerCore;
    tiling.nNum = nNum;
    tiling.useNesterov = use_nesterov ? 1 : 0;
    tiling.pad0 = 0;
    tiling.beta1 = static_cast<float>(beta1);
    tiling.beta2 = static_cast<float>(beta2);
    tiling.oneMinusBeta1 = static_cast<float>(1.0 - beta1);
    tiling.oneMinusBeta2 = static_cast<float>(1.0 - beta2);
    const double invBias1 = 1.0 / (1.0 - beta1_power);
    const double invBias2 = 1.0 / (1.0 - beta2_power);
    tiling.invBias1 = static_cast<float>(invBias1);
    tiling.invBias2 = static_cast<float>(invBias2);
    tiling.lr = static_cast<float>(lr);
    tiling.negLr = static_cast<float>(-lr);
    tiling.epsilon = static_cast<float>(epsilon);
    tiling.coeffGradForNesterov = static_cast<float>((1.0 - beta1) / (1.0 - beta1_power));

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(AdamWKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(tilingCpu.data_ptr(), &tiling, sizeof(AdamWKernelTiling));
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    LaunchFn launch = nullptr;
    if (var.scalar_type() == at::kFloat) {
        launch = adamw_apply_do_fp32;
    } else if (var.scalar_type() == at::kHalf) {
        launch = adamw_apply_do_fp16;
    } else if (var.scalar_type() == at::kBFloat16) {
        launch = adamw_apply_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(var.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(m.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(v.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(grad.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(varOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(mOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(vOut.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return pybind11::make_tuple(varOut, mOut, vOut);
}

}  // namespace adamw_ext

PYBIND11_MODULE(_adamw_ext, m)
{
    m.doc() = "adamw extension";
    m.def("run_adamw", &adamw_ext::run_adamw, "AdamW kernel: returns (var_out, m_out, v_out)");
}
