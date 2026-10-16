#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "add_tiling.h"

extern "C" void add_alpha_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *out,
    uint8_t *tiling);

extern "C" void add_alpha_do_fp16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *out,
    uint8_t *tiling);

extern "C" void add_alpha_do_bf16(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *out,
    uint8_t *tiling);

namespace add_alpha_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

at::Tensor run_add_alpha(
    const at::Tensor &x,
    const at::Tensor &y,
    double alpha)
{
    TORCH_CHECK(x.dim() == 1, "x must be 1D (pre-flattened)");
    TORCH_CHECK(y.dim() == 1, "y must be 1D (pre-flattened)");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same number of elements");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "x and y must have the same dtype");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    const auto n = static_cast<int32_t>(x.numel());

    // Choose tile size based on dtype
    int32_t tileSize = DEFAULT_TILE_SIZE;
    if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
        tileSize = DEFAULT_TILE_SIZE_SMALL;
    }

    const int32_t totalTiles = (n + tileSize - 1) / tileSize;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, totalTiles);
    const int32_t tasksPerCore = (totalTiles + usedCoreNum - 1) / usedCoreNum;

    at::Tensor out = at::empty_like(x);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(AddKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<AddKernelTiling *>(tilingCpu.data_ptr());
    tiling->N = n;
    tiling->tileSize = tileSize;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->totalTiles = totalTiles;
    tiling->alpha = static_cast<float>(alpha);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        launch = add_alpha_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = add_alpha_do_fp16;
    } else if (x.scalar_type() == at::kBFloat16) {
        launch = add_alpha_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(out.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return out;
}

}  // namespace add_alpha_ext

PYBIND11_MODULE(_add_ext, m)
{
    m.doc() = "add_alpha extension";
    m.def("run_add_alpha", &add_alpha_ext::run_add_alpha, "");
}
