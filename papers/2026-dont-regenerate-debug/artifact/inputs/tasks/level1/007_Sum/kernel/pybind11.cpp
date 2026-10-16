#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "sum_tiling.h"

extern "C" void sum_small_n_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void sum_small_n_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void sum_small_n_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void sum_large_n_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void sum_large_n_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void sum_large_n_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);

namespace sum_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);

at::Tensor run_sum(const at::Tensor &x)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const auto m = static_cast<int32_t>(x.sizes()[0]);
    const auto n = static_cast<int32_t>(x.sizes()[1]);

    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    const int32_t tileN = DEFAULT_TILE_N;
    const int32_t nTiles = (n + tileN - 1) / tileN;
    const int32_t lastTileValid = n - (nTiles - 1) * tileN;

    at::Tensor y = at::empty({m, 1}, x.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(SumKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<SumKernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->tileN = tileN;
    tiling->nTiles = nTiles;
    tiling->lastTileValid = lastTileValid;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;

    const bool isSmallN = (n <= DEFAULT_TILE_N);

    if (x.scalar_type() == at::kFloat) {
        launch = isSmallN ? sum_small_n_do_fp32 : sum_large_n_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = isSmallN ? sum_small_n_do_fp16 : sum_large_n_do_fp16;
    } else {
        launch = isSmallN ? sum_small_n_do_bf16 : sum_large_n_do_bf16;
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));
    return y;
}

}  // namespace sum_ext

PYBIND11_MODULE(_sum_ext, m)
{
    m.doc() = "sum extension";
    m.def("run_sum", &sum_ext::run_sum, "");
}
