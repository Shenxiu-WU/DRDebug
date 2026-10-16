#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "moe_init_routing_tiling.h"

extern "C" void moe_init_routing_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *expandedRowIdx,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void moe_init_routing_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *expandedRowIdx,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void moe_init_routing_bf16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *expandedRowIdx,
    uint8_t *y,
    uint8_t *tiling);

namespace moe_init_routing_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

inline LaunchFn ResolveLaunchFn(at::ScalarType dtype)
{
    if (dtype == at::kHalf) {
        return moe_init_routing_fp16_do;
    }
    if (dtype == at::kFloat) {
        return moe_init_routing_fp32_do;
    }
    if (dtype == at::kBFloat16) {
        return moe_init_routing_bf16_do;
    }
    TORCH_CHECK(false, "x must be float16 / bfloat16 / float32");
    return nullptr;
}

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline int32_t ChooseBlockM(int32_t m)
{
    int32_t blockM = 32;
    if (m < 32) {
        blockM = 16;
    }
    if (m < 16) {
        blockM = 8;
    }
    if (m < 8) {
        blockM = 4;
    }
    if (m < 4) {
        blockM = 2;
    }
    if (m < 2) {
        blockM = 1;
    }
    return blockM;
}

at::Tensor run_moe_init_routing(
    const at::Tensor &x,
    const at::Tensor &expanded_row_idx,
    int64_t H,
    int64_t XStride,
    int64_t YStride)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(expanded_row_idx.device().is_privateuseone(), "expanded_row_idx must be on NPU");
    TORCH_CHECK(x.dim() == 2, "x must be [N, XStride]");
    TORCH_CHECK(expanded_row_idx.dim() == 1, "expanded_row_idx must be 1D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(expanded_row_idx.is_contiguous(), "expanded_row_idx must be contiguous");
    TORCH_CHECK(expanded_row_idx.scalar_type() == at::kInt, "expanded_row_idx must be int32");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat ||
            x.scalar_type() == at::kBFloat16,
        "x dtype must be float16 / bfloat16 / float32");

    const int32_t N32 = static_cast<int32_t>(x.size(0));
    const int32_t XStride32 = static_cast<int32_t>(x.size(1));
    TORCH_CHECK(XStride32 == static_cast<int32_t>(XStride),
                "XStride must match x.size(1)");
    const int32_t M32 = static_cast<int32_t>(expanded_row_idx.size(0));
    const int32_t H32 = static_cast<int32_t>(H);
    const int32_t YStride32 = static_cast<int32_t>(YStride);

    TORCH_CHECK(H32 > 0, "H must be positive");
    TORCH_CHECK(H32 <= XStride32, "H must not exceed XStride");
    TORCH_CHECK(H32 <= YStride32, "H must not exceed YStride");

    at::Tensor y = at::zeros({M32, YStride32}, x.options());

    const int32_t blockM = ChooseBlockM(std::max(M32, 1));
    const int32_t mNum = CeilDivI32(std::max(M32, 1), blockM);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(mNum, 1));
    const int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);
    const int32_t subBlockM = std::max(blockM / DEFAULT_VEC_NUM, 1);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(MoeInitRoutingKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<MoeInitRoutingKernelTiling *>(tilingCpu.data_ptr());
    tiling->M = M32;
    tiling->N = N32;
    tiling->H = H32;
    tiling->XStride = XStride32;
    tiling->YStride = YStride32;
    tiling->blockM = blockM;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockM = subBlockM;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = ResolveLaunchFn(x.scalar_type());
    launch(
        static_cast<uint32_t>(usedCoreNum),
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(expanded_row_idx.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return y;
}

} // namespace moe_init_routing_ext

PYBIND11_MODULE(_moe_init_routing_ext, m)
{
    m.doc() = "MoE init routing AscendC extension";
    m.def("run_moe_init_routing", &moe_init_routing_ext::run_moe_init_routing, "");
}
