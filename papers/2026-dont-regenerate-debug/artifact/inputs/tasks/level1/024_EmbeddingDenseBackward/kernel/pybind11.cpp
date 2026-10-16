#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "embedding_dense_backward_tiling.h"

extern "C" void embedding_dense_backward_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *indices, uint8_t *gradWeight, uint8_t *tiling);

extern "C" void embedding_dense_backward_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *indices, uint8_t *gradWeight, uint8_t *tiling);

extern "C" void embedding_dense_backward_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *gradOutput, uint8_t *indices, uint8_t *gradWeight, uint8_t *tiling);

namespace embedding_dense_backward_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::tuple run_embedding_dense_backward(
    const at::Tensor &gradOutput,
    const at::Tensor &indices,
    int64_t numWeights,
    int64_t paddingIdx,
    bool scaleGradByFreq)
{
    TORCH_CHECK(gradOutput.dim() >= 2, "gradOutput must be at least 2D");
    TORCH_CHECK(indices.dim() >= 1, "indices must be at least 1D");
    TORCH_CHECK(
        gradOutput.scalar_type() == at::kFloat ||
        gradOutput.scalar_type() == at::kHalf ||
        gradOutput.scalar_type() == at::kBFloat16,
        "gradOutput must be float16, float32, or bfloat16");
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
        "indices must be int32 or int64");
    TORCH_CHECK(gradOutput.is_contiguous(), "gradOutput must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    const auto totalIndices = static_cast<int32_t>(indices.numel());
    const auto D = static_cast<int32_t>(gradOutput.sizes()[gradOutput.dim() - 1]);
    const auto numWeightsInt = static_cast<int32_t>(numWeights);

    const int32_t dNum = (D + DEFAULT_BLOCK_D - 1) / DEFAULT_BLOCK_D;
    const int32_t wNum = (numWeightsInt + DEFAULT_BLOCK_WEIGHTS - 1) / DEFAULT_BLOCK_WEIGHTS;
    const int32_t totalBlocks = wNum * dNum;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, totalBlocks);
    const int32_t tasksPerCore = (totalBlocks + usedCoreNum - 1) / usedCoreNum;

    at::Tensor gradWeight = at::zeros({numWeightsInt, D}, gradOutput.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(EmbeddingDenseBackwardTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<EmbeddingDenseBackwardTiling *>(tilingCpu.data_ptr());
    tiling->totalIndices = totalIndices;
    tiling->D = D;
    tiling->numWeights = numWeightsInt;
    tiling->paddingIdx = static_cast<int32_t>(paddingIdx);
    tiling->scaleGradByFreq = scaleGradByFreq ? 1 : 0;
    tiling->blockWeights = DEFAULT_BLOCK_WEIGHTS;
    tiling->blockD = DEFAULT_BLOCK_D;
    tiling->blockI = DEFAULT_BLOCK_I;
    tiling->totalBlocks = totalBlocks;
    tiling->dNum = dNum;
    tiling->tasksPerCore = tasksPerCore;
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (gradOutput.scalar_type() == at::kFloat) {
        launch = embedding_dense_backward_do_fp32;
    } else if (gradOutput.scalar_type() == at::kHalf) {
        launch = embedding_dense_backward_do_fp16;
    } else if (gradOutput.scalar_type() == at::kBFloat16) {
        launch = embedding_dense_backward_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(gradOutput.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(indices.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gradWeight.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return pybind11::make_tuple(gradWeight);
}

} // namespace embedding_dense_backward_ext

PYBIND11_MODULE(_embedding_dense_backward_ext, m)
{
    m.doc() = "embedding_dense_backward extension";
    m.def("run_embedding_dense_backward", &embedding_dense_backward_ext::run_embedding_dense_backward, "");
}
