#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "avg_pool3_d_tiling.h"

extern "C" void avg_pool3_d_generic_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling);
extern "C" void avg_pool3_d_reduce_d_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling);
extern "C" void avg_pool3_d_split_c_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling);
extern "C" void avg_pool3_d_split_w_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling);
extern "C" void avg_pool3_d_multi_w_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling);

namespace avg_pool3_d_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);

constexpr int32_t SPLIT_MODE_C = 1;
constexpr int32_t SPLIT_MODE_W = 2;
constexpr int32_t SPLIT_MODE_MULTI_W = 3;

constexpr int32_t IMPL_GENERIC = 0;
constexpr int32_t IMPL_REDUCE_D = 1;
constexpr int32_t IMPL_SPLIT_C = 2;
constexpr int32_t IMPL_SPLIT_W = 3;
constexpr int32_t IMPL_MULTI_W = 4;

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

int32_t ChooseBlockM(int32_t mOut)
{
    for (int32_t candidate : {64, 32, 16, 8, 4, 2}) {
        if (candidate <= mOut && mOut % candidate == 0) {
            return candidate;
        }
    }
    TORCH_CHECK(false, "Unsupported output spatial size: M_out=", mOut);
}

LaunchFn ResolveLaunchFn(int32_t implMode)
{
    switch (implMode) {
        case IMPL_REDUCE_D:
            return avg_pool3_d_reduce_d_do;
        case IMPL_SPLIT_C:
            return avg_pool3_d_split_c_do;
        case IMPL_SPLIT_W:
            return avg_pool3_d_split_w_do;
        case IMPL_MULTI_W:
            return avg_pool3_d_multi_w_do;
        case IMPL_GENERIC:
        default:
            return avg_pool3_d_generic_do;
    }
}

at::Tensor run_avg_pool3_d(
    const at::Tensor &xFlat,
    int64_t n,
    int64_t c,
    int64_t d,
    int64_t h,
    int64_t w,
    int64_t od,
    int64_t oh,
    int64_t ow,
    int64_t kD,
    int64_t kH,
    int64_t kW,
    int64_t sD,
    int64_t sH,
    int64_t sW,
    int64_t pD,
    int64_t pH,
    int64_t pW,
    int64_t countIncludePad,
    int64_t divisorOverride,
    int64_t splitMode,
    int64_t blockC,
    int64_t splitWTileKw,
    int64_t multiWWindow,
    int64_t implMode)
{
    TORCH_CHECK(xFlat.dim() == 2, "xFlat must be 2D");
    TORCH_CHECK(xFlat.is_contiguous(), "xFlat must be contiguous");
    TORCH_CHECK(xFlat.scalar_type() == at::kFloat, "AscendC kernel currently supports float32");

    const int32_t n32 = static_cast<int32_t>(n);
    const int32_t c32 = static_cast<int32_t>(c);
    const int32_t d32 = static_cast<int32_t>(d);
    const int32_t h32 = static_cast<int32_t>(h);
    const int32_t w32 = static_cast<int32_t>(w);
    const int32_t od32 = static_cast<int32_t>(od);
    const int32_t oh32 = static_cast<int32_t>(oh);
    const int32_t ow32 = static_cast<int32_t>(ow);

    const int32_t inSpatial = d32 * h32 * w32;
    const int32_t outSpatial = od32 * oh32 * ow32;
    const int32_t mOut = n32 * outSpatial;

    TORCH_CHECK(xFlat.size(0) == static_cast<int64_t>(n32) * inSpatial, "xFlat row size mismatch");
    TORCH_CHECK(xFlat.size(1) == c32, "xFlat channel size mismatch");

    const int32_t blockM = ChooseBlockM(mOut);
    const int32_t subBlockM = blockM / DEFAULT_VEC_NUM;
    const int32_t mNum = mOut / blockM;
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);

    const int32_t splitMode32 = static_cast<int32_t>(splitMode);
    int32_t blockC32 = static_cast<int32_t>(blockC);
    if (splitMode32 != SPLIT_MODE_C) {
        blockC32 = 0;
    }

    int32_t cNum = 1;
    if (blockC32 > 0) {
        TORCH_CHECK(c32 % blockC32 == 0, "blockC must divide C");
        cNum = c32 / blockC32;
    }

    int32_t splitWTileKw32 = static_cast<int32_t>(splitWTileKw);
    if (splitMode32 != SPLIT_MODE_W || splitWTileKw32 <= 0) {
        splitWTileKw32 = 0;
    }

    int32_t multiWWindow32 = static_cast<int32_t>(multiWWindow);
    if (splitMode32 != SPLIT_MODE_MULTI_W || multiWWindow32 <= 1) {
        multiWWindow32 = 1;
    }

    at::Tensor yFlat = at::empty({mOut, c32}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor tilingCpu = at::empty({static_cast<long>(sizeof(AvgPool3DKernelTiling))}, at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<AvgPool3DKernelTiling *>(tilingCpu.data_ptr());

    tiling->N = n32;
    tiling->C = c32;
    tiling->D = d32;
    tiling->H = h32;
    tiling->W = w32;

    tiling->OD = od32;
    tiling->OH = oh32;
    tiling->OW = ow32;

    tiling->kD = static_cast<int32_t>(kD);
    tiling->kH = static_cast<int32_t>(kH);
    tiling->kW = static_cast<int32_t>(kW);

    tiling->sD = static_cast<int32_t>(sD);
    tiling->sH = static_cast<int32_t>(sH);
    tiling->sW = static_cast<int32_t>(sW);

    tiling->pD = static_cast<int32_t>(pD);
    tiling->pH = static_cast<int32_t>(pH);
    tiling->pW = static_cast<int32_t>(pW);

    tiling->countIncludePad = static_cast<int32_t>(countIncludePad);
    tiling->divisorOverride = static_cast<int32_t>(divisorOverride);

    tiling->splitMode = splitMode32;
    tiling->blockC = blockC32;
    tiling->splitWTileKw = splitWTileKw32;
    tiling->multiWWindow = multiWWindow32;

    tiling->blockM = blockM;
    tiling->subBlockM = subBlockM;
    tiling->mNum = mNum;
    tiling->outSpatial = outSpatial;
    tiling->inSpatial = inSpatial;
    tiling->hw = h32 * w32;

    tiling->cNum = cNum;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    tiling->vectorLen = std::min<int32_t>(c32, 256);
    tiling->reserved0 = static_cast<int32_t>(implMode);
    tiling->reserved1 = 0;

    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    tiling->launchBlocks = static_cast<int32_t>(blockDim);

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    LaunchFn launch = ResolveLaunchFn(static_cast<int32_t>(implMode));
    launch(
        blockDim,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(xFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(yFlat.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return yFlat;
}

} // namespace avg_pool3_d_ext

PYBIND11_MODULE(_current_task_ext, m)
{
    m.doc() = "current_task avg_pool3_d AscendC extension";
    m.def("run_avg_pool3_d", &avg_pool3_d_ext::run_avg_pool3_d, "");
}
