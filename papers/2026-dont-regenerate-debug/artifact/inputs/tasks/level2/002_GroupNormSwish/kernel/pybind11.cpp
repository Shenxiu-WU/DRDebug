#include <algorithm>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "group_norm_swish_tiling.h"

extern "C" void group_norm_swish_do_fp32(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight, uint8_t *bias,
    uint8_t *output, uint8_t *mean, uint8_t *rstd, uint8_t *tiling);

extern "C" void group_norm_swish_do_fp16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight, uint8_t *bias,
    uint8_t *output, uint8_t *mean, uint8_t *rstd, uint8_t *tiling);

extern "C" void group_norm_swish_do_bf16(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight, uint8_t *bias,
    uint8_t *output, uint8_t *mean, uint8_t *rstd, uint8_t *tiling);

namespace group_norm_swish_ext {

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *,
                          uint8_t *, uint8_t *, uint8_t *, uint8_t *);

pybind11::tuple run_group_norm_swish(
    const at::Tensor &input,
    int64_t num_groups,
    const at::Tensor &weight,
    const at::Tensor &bias,
    double eps,
    double swish_scale)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    TORCH_CHECK(
        input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
        "input must be float16, float32, or bfloat16");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input");
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(), "bias dtype must match input");

    int32_t N = static_cast<int32_t>(input.size(0));
    int32_t C = static_cast<int32_t>(input.size(1));
    int64_t spatial_size = input.numel() / (static_cast<int64_t>(N) * C);
    int32_t group_size = C / static_cast<int32_t>(num_groups);
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");
    TORCH_CHECK(weight.numel() == C, "weight shape mismatch");
    TORCH_CHECK(bias.numel() == C, "bias shape mismatch");

    int32_t M = N * static_cast<int32_t>(num_groups);
    int32_t usedCoreNum = std::min<int32_t>(kNumPhysicalCores, M);
    int32_t tasksPerCore = (M + usedCoreNum - 1) / usedCoreNum;

    at::Tensor output = at::empty_like(input);
    at::Tensor mean = at::empty({N, static_cast<int64_t>(num_groups)}, input.options());
    at::Tensor rstd = at::empty({N, static_cast<int64_t>(num_groups)}, input.options());

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(GroupNormSwishTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<GroupNormSwishTiling *>(tilingCpu.data_ptr());
    tiling->N = N;
    tiling->C = C;
    tiling->spatialSize = static_cast<int32_t>(spatial_size);
    tiling->numGroups = static_cast<int32_t>(num_groups);
    tiling->groupSize = group_size;
    tiling->M = M;
    tiling->K = group_size * static_cast<int32_t>(spatial_size);
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->eps = static_cast<float>(eps);
    tiling->invK = 1.0f / static_cast<float>(tiling->K);
    tiling->swishScale = static_cast<float>(swish_scale);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    LaunchFn launch = nullptr;
    if (input.scalar_type() == at::kFloat) {
        launch = group_norm_swish_do_fp32;
    } else if (input.scalar_type() == at::kHalf) {
        launch = group_norm_swish_do_fp16;
    } else if (input.scalar_type() == at::kBFloat16) {
        launch = group_norm_swish_do_bf16;
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(input.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(weight.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(bias.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(output.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(mean.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(rstd.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return pybind11::make_tuple(output, mean, rstd);
}

}  // namespace group_norm_swish_ext

PYBIND11_MODULE(_group_norm_swish_ext, m)
{
    m.doc() = "group_norm_swish extension";
    m.def("run_group_norm_swish", &group_norm_swish_ext::run_group_norm_swish, "");
}
