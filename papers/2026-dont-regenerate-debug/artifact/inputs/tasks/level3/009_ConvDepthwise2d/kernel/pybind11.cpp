#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "depthwise_conv2d_tiling.h"

extern "C" void depthwise_conv2d_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *w,
    uint8_t *bias,
    uint8_t *y,
    uint8_t *tiling);

namespace depthwise_conv2d_ext {

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

int32_t ChooseBlockM(int32_t mOut)
{
    for (int32_t candidate : {32, 16, 8, 4, 2}) {
        if (candidate <= mOut && mOut % candidate == 0) {
            return candidate;
        }
    }
    return 1;
}

at::Tensor run_depthwise_conv2d(
    const at::Tensor &xFlat,
    const at::Tensor &weightFilter,
    const at::Tensor &biasVec,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    int64_t oh,
    int64_t ow,
    int64_t kH,
    int64_t kW,
    int64_t sH,
    int64_t sW,
    int64_t pH,
    int64_t pW,
    int64_t hasBias)
{
    TORCH_CHECK(xFlat.dim() == 2, "xFlat must be 2D (N*H*W, C)");
    TORCH_CHECK(weightFilter.dim() == 2, "weightFilter must be 2D (kH*kW, C)");
    TORCH_CHECK(biasVec.dim() == 1, "biasVec must be 1D (C,)");
    TORCH_CHECK(xFlat.is_contiguous(), "xFlat must be contiguous");
    TORCH_CHECK(weightFilter.is_contiguous(), "weightFilter must be contiguous");
    TORCH_CHECK(biasVec.is_contiguous(), "biasVec must be contiguous");
    TORCH_CHECK(xFlat.scalar_type() == at::kFloat, "depthwise_conv2d only supports float32");

    const int32_t n32 = static_cast<int32_t>(n);
    const int32_t c32 = static_cast<int32_t>(c);
    const int32_t h32 = static_cast<int32_t>(h);
    const int32_t w32 = static_cast<int32_t>(w);
    const int32_t oh32 = static_cast<int32_t>(oh);
    const int32_t ow32 = static_cast<int32_t>(ow);
    const int32_t kh32 = static_cast<int32_t>(kH);
    const int32_t kw32 = static_cast<int32_t>(kW);

    const int32_t inSpatial = h32 * w32;
    const int32_t outSpatial = oh32 * ow32;
    const int32_t mOut = n32 * outSpatial;
    const int32_t weightCount = kh32 * kw32 * c32;

    TORCH_CHECK(xFlat.size(0) == static_cast<int64_t>(n32) * inSpatial, "xFlat row count mismatch");
    TORCH_CHECK(xFlat.size(1) == c32, "xFlat channel size mismatch");
    TORCH_CHECK(weightFilter.size(0) == kh32 * kw32, "weightFilter row count mismatch");
    TORCH_CHECK(weightFilter.size(1) == c32, "weightFilter channel size mismatch");
    TORCH_CHECK(biasVec.size(0) == c32, "biasVec size mismatch");

    int32_t blockM = ChooseBlockM(mOut);
    if (blockM <= 0) {
        blockM = 1;
    }
    int32_t subBlockM = blockM / DEFAULT_VEC_NUM;
    if (subBlockM <= 0) {
        subBlockM = 1;
    }
    const int32_t mNum = (mOut + blockM - 1) / blockM;
    int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    const int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);

    at::Tensor yFlat = at::empty({mOut, c32}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(DepthwiseConv2DKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<DepthwiseConv2DKernelTiling *>(tilingCpu.data_ptr());

    tiling->N = n32;
    tiling->C = c32;
    tiling->H = h32;
    tiling->W = w32;

    tiling->OH = oh32;
    tiling->OW = ow32;

    tiling->kH = kh32;
    tiling->kW = kw32;

    tiling->sH = static_cast<int32_t>(sH);
    tiling->sW = static_cast<int32_t>(sW);

    tiling->pH = static_cast<int32_t>(pH);
    tiling->pW = static_cast<int32_t>(pW);

    tiling->hasBias = static_cast<int32_t>(hasBias);

    tiling->inSpatial = inSpatial;
    tiling->outSpatial = outSpatial;
    tiling->weightCount = weightCount;
    tiling->mOut = mOut;

    tiling->blockM = blockM;
    tiling->subBlockM = subBlockM;
    tiling->mNum = mNum;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    tiling->launchBlocks = static_cast<int32_t>(blockDim);

    tiling->reserved0 = 0;
    tiling->reserved1 = 0;
    tiling->reserved2 = 0;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    depthwise_conv2d_do(
        blockDim,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(xFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(weightFilter.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(biasVec.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(yFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return yFlat;
}

} // namespace depthwise_conv2d_ext

PYBIND11_MODULE(_current_task_ext, m)
{
    m.doc() = "current_task depthwise_conv2d AscendC extension";
    m.def("run_depthwise_conv2d", &depthwise_conv2d_ext::run_depthwise_conv2d, "");
}
