#include <algorithm>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "layer_norm_tiling.h"

// Onepass launch functions
extern "C" void layer_norm_onepass_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void layer_norm_onepass_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void layer_norm_onepass_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

// Twopass launch functions
extern "C" void layer_norm_twopass_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void layer_norm_twopass_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void layer_norm_twopass_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

namespace layer_norm_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::tuple run_layer_norm(
    const at::Tensor &x, const at::Tensor &gamma, const at::Tensor &beta, int64_t hasBias, double eps)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [N]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(), "gamma dtype must match x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x.sizes()[1] == gamma.sizes()[0], "gamma shape mismatch");

    const auto m = static_cast<int32_t>(x.sizes()[0]);
    const auto n = static_cast<int32_t>(x.sizes()[1]);

    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;
    const int32_t subBlockM = DEFAULT_BLOCK_M / DEFAULT_VEC_NUM;
    const int32_t nChunks = (n + DEFAULT_BLOCK_N_CHUNK - 1) / DEFAULT_BLOCK_N_CHUNK;

    at::Tensor y = at::empty_like(x);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(LayerNormKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<LayerNormKernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockM = subBlockM;
    tiling->eps = static_cast<float>(eps);
    tiling->invN = 1.0f / static_cast<float>(n);
    tiling->hasBias = static_cast<int32_t>(hasBias);
    tiling->nChunks = nChunks;
    tiling->blockNChunk = DEFAULT_BLOCK_N_CHUNK;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    bool useTwopass = (n > 4096);

    if (x.scalar_type() == at::kFloat) {
        launch = useTwopass ? layer_norm_twopass_do_fp32 : layer_norm_onepass_do_fp32;
    } else if (x.scalar_type() == at::kHalf) {
        launch = useTwopass ? layer_norm_twopass_do_fp16 : layer_norm_onepass_do_fp16;
    } else {
        launch = useTwopass ? layer_norm_twopass_do_bf16 : layer_norm_onepass_do_bf16;
    }

    uint8_t *xPtr = static_cast<uint8_t *>(const_cast<void *>(x.storage().data()));
    uint8_t *gammaPtr = static_cast<uint8_t *>(const_cast<void *>(gamma.storage().data()));
    uint8_t *betaPtr = static_cast<uint8_t *>(const_cast<void *>(beta.storage().data()));
    uint8_t *yPtr = static_cast<uint8_t *>(const_cast<void *>(y.storage().data()));
    uint8_t *tilingPtr = static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data()));

    launch(usedCoreNum, aclStream, xPtr, gammaPtr, betaPtr, yPtr, tilingPtr);

    return pybind11::make_tuple(y);
}

}  // namespace layer_norm_ext

PYBIND11_MODULE(_layer_norm_ext, m)
{
    m.doc() = "layer_norm extension";
    m.def("run_layer_norm", &layer_norm_ext::run_layer_norm, "");
}
