#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "conv1d_tiling.h"

extern "C" void conv1d_do(uint32_t blockDim, void *stream, uint8_t *x,
                          uint8_t *weight, uint8_t *bias, uint8_t *y,
                          uint8_t *tiling);

namespace conv1d_ext {

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

at::Tensor run_conv1d(
    const at::Tensor &xNlc,
    const at::Tensor &wOki,
    const at::Tensor &biasFlat,
    int64_t n,
    int64_t cIn,
    int64_t cOut,
    int64_t lIn,
    int64_t lOut,
    int64_t kSize,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    int64_t hasBias)
{
    TORCH_CHECK(xNlc.dim() == 2, "xNlc must be 2D (N*L_in, C_in)");
    TORCH_CHECK(xNlc.is_contiguous(), "xNlc must be contiguous");
    TORCH_CHECK(xNlc.scalar_type() == at::kFloat, "kernel currently supports float32");
    TORCH_CHECK(wOki.is_contiguous(), "wOki must be contiguous");
    TORCH_CHECK(wOki.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(biasFlat.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(biasFlat.scalar_type() == at::kFloat, "bias must be float32");

    const int32_t n32 = static_cast<int32_t>(n);
    const int32_t cIn32 = static_cast<int32_t>(cIn);
    const int32_t cOut32 = static_cast<int32_t>(cOut);
    const int32_t lIn32 = static_cast<int32_t>(lIn);
    const int32_t lOut32 = static_cast<int32_t>(lOut);
    const int32_t k32 = static_cast<int32_t>(kSize);
    const int32_t stride32 = static_cast<int32_t>(stride);
    const int32_t pad32 = static_cast<int32_t>(padding);
    const int32_t dil32 = static_cast<int32_t>(dilation);
    const int32_t g32 = static_cast<int32_t>(groups);

    TORCH_CHECK(g32 > 0 && cIn32 % g32 == 0, "groups must divide C_in");
    TORCH_CHECK(cOut32 % g32 == 0, "groups must divide C_out");

    const int32_t cInPerGroup = cIn32 / g32;
    const int32_t cOutPerGroup = cOut32 / g32;

    TORCH_CHECK(xNlc.size(0) == static_cast<int64_t>(n32) * lIn32,
                "xNlc row count mismatch with n*L_in");
    TORCH_CHECK(xNlc.size(1) == cIn32, "xNlc col count mismatch with C_in");
    TORCH_CHECK(wOki.numel() == static_cast<int64_t>(cOut32) * k32 * cInPerGroup,
                "weight numel mismatch");
    TORCH_CHECK(biasFlat.numel() == cOut32 || biasFlat.numel() == 0,
                "bias numel mismatch");

    const int32_t totalBlocks = n32 * lOut32;
    int32_t usedCoreNum = std::min<int32_t>(CONV1D_DEFAULT_NUM_PHYSICAL_CORES, totalBlocks);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    const int32_t tasksPerCore = CeilDivI32(totalBlocks, usedCoreNum);

    at::Tensor yNlo = at::empty({static_cast<int64_t>(n32) * lOut32, cOut32},
                                 at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor tilingCpu = at::empty({static_cast<long>(sizeof(Conv1DKernelTiling))},
                                      at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<Conv1DKernelTiling *>(tilingCpu.data_ptr());

    tiling->N = n32;
    tiling->cIn = cIn32;
    tiling->cOut = cOut32;
    tiling->lIn = lIn32;
    tiling->lOut = lOut32;

    tiling->kSize = k32;
    tiling->stride = stride32;
    tiling->padding = pad32;
    tiling->dilation = dil32;
    tiling->groups = g32;

    tiling->cInPerGroup = cInPerGroup;
    tiling->cOutPerGroup = cOutPerGroup;

    tiling->hasBias = static_cast<int32_t>(hasBias);
    tiling->totalBlocks = totalBlocks;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    tiling->reserved0 = 0;
    tiling->reserved1 = 0;
    tiling->reserved2 = 0;
    tiling->reserved3 = 0;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    conv1d_do(
        blockDim,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(xNlc.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(wOki.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(biasFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(yNlo.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data()))
    );

    return yNlo;
}

} // namespace conv1d_ext

PYBIND11_MODULE(_conv1d_ext, m)
{
    m.doc() = "AscendC Conv1D extension";
    m.def("run_conv1d", &conv1d_ext::run_conv1d, "");
}
